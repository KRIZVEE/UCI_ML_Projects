import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Linear Regression
# ML Pipeline

np.random.seed(42)

# data collection
df_sc = pd.read_csv('train.csv', sep = ',')
print('dataframe shape : ', df_sc.shape)
print('dataframe head : ',df_sc.head())

# data preprocessing [categorical_data--> numerical data | label can be done to binary classification]

# data preparation
orig_train, test = train_test_split(df_sc.values, test_size = 0.2)
print('orig_train shape : ', orig_train.shape,
      'test shape : ', test.shape)
orig_train_X = orig_train[:,:-1]
orig_train_y = orig_train[:,-1]
test_X = test[:,:-1]
test_y = test[:,-1]
print('--------')
train, val = train_test_split(orig_train, test_size = 0.2)
print('train shape : ',train.shape,
      'val shape : ',val.shape)
train_X = orig_train[:,:-1]
train_y = orig_train[:,-1]
val_y = test[:,-1]
val_X = test[:,:-1]
# build model
model_lr = LinearRegression()
model_svr = SVR()
model_rfr = RandomForestRegressor()

# train model
model_lr = model_lr.fit(train_X, train_y)
model_svr = model_svr.fit(train_X, train_y)
model_rfr = model_rfr.fit(train_X, train_y)
# exit(1)
# validate model
predict_lr = model_lr.predict(val_X)
predict_svr = model_svr.predict(val_X)
predict_rfr = model_rfr.predict(val_X)

# model selection based on validation AUC score
print('LR val MSE score : ', mean_squared_error(val_y, predict_lr),
      'SVR val MSE score : ', mean_squared_error(val_y, predict_svr),
      'RFR val MSE score : ', mean_squared_error(val_y, predict_rfr))
# test model and report accuracy
model = RandomForestRegressor()
model = model.fit(orig_train_X,orig_train_y)
predict = model.predict(test_X)
print(' RFR test MSE score : ', mean_squared_error(test_y, predict))