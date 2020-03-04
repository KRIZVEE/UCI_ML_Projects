import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

# ML Pipeline
np.random.seed(42)

# data collection
df_wifi = pd.read_csv('wifi_localization.txt', sep = '\t')
headers = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'label']
df_wifi.columns = headers
print('dataframe shape',df_wifi.shape)
# print(df_wifi.head())

# data preprocessing [categorical_data--> numerical data | label can be done to binary classification]
pos_index = df_wifi['label'] >= 2
neg_index = df_wifi['label'] < 2

df_wifi.loc[pos_index,'label'] = 1
df_wifi.loc[neg_index,'label'] = 0
# print(df_wifi.head(20))
# print(df_wifi.tail(20))

# data preparation
orig_train, test = train_test_split(df_wifi.values, test_size = 0.2)
print('orig_train shape : ', orig_train.shape)
print('test shape : ', test.shape)
orig_train_X = orig_train[:,:-1]
orig_train_y = orig_train[:,-1]
test_X = test[:,:-1]
test_y = test[:,-1]
print()
train, val = train_test_split(orig_train, test_size = 0.2)
print('train shape : ', train.shape)
print('val shape : ', val.shape)
train_X = train[:,:-1]
train_y = train[:,-1]
val_X = val[:,:-1]
val_y = val[:,-1]
print()

# exit(1)

# build model
model_lr = LogisticRegression(max_iter = 300)
model_sgd = SGDClassifier(loss = "modified_huber")
model_svc = SVC(probability = True)
model_knn = KNeighborsClassifier()
model_rnc = RadiusNeighborsClassifier(radius = 11.0)
model_gnb = GaussianNB()
model_dtc = DecisionTreeClassifier()
model_abc = AdaBoostClassifier()
model_rfc = RandomForestClassifier()

estimators = [
              ('LR', LogisticRegression(max_iter = 300)),
              ('SVM', SVC()),
              ('RF', RandomForestClassifier())
              ]

model_sc = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression(max_iter = 300))
model_mlpc = MLPClassifier(max_iter = 300)

# train model
model_lr = model_lr.fit(train_X, train_y)
model_sgd = model_sgd.fit(train_X, train_y)
model_svc = model_svc.fit(train_X, train_y)
model_knn = model_knn.fit(train_X, train_y)
model_rnc = model_rnc.fit(train_X, train_y)
model_gnb = model_gnb.fit(train_X, train_y)
model_dtc = model_dtc.fit(train_X, train_y)
model_abc = model_abc.fit(train_X, train_y)
model_rfc = model_rfc.fit(train_X, train_y)
model_sc = model_sc.fit(train_X, train_y)
model_mlpc = model_mlpc.fit(train_X, train_y)

# validate model
pred_lr = model_lr.predict_proba(val_X)[:, 1]
pred_sgd = model_sgd.predict_proba(val_X)[:, 1]
pred_svc = model_svc.predict_proba(val_X)[:, 1]
pred_knn = model_knn.predict_proba(val_X)[:, 1]
pred_rnc = model_rnc.predict_proba(val_X)[:, 1]
pred_gnb = model_gnb.predict_proba(val_X)[:, 1]
pred_dtc = model_dtc.predict_proba(val_X)[:, 1]
pred_abc = model_abc.predict_proba(val_X)[:, 1]
pred_rfc = model_rfc.predict_proba(val_X)[:, 1]
pred_sc = model_sc.predict_proba(val_X)[:, 1]
pred_mlpc = model_mlpc.predict_proba(val_X)[:, 1]

# model selection based on validation AUC score
print('LR val AUC score : ', roc_auc_score(val_y, pred_lr))
print('SGD val AUC score : ', roc_auc_score(val_y, pred_sgd))
print('SVC val AUC score : ', roc_auc_score(val_y, pred_svc))
print('KNN val AUC score : ', roc_auc_score(val_y, pred_knn))
print('RNC val AUC score : ', roc_auc_score(val_y, pred_rnc))
print('GNB val AUC score : ', roc_auc_score(val_y, pred_gnb))
print('DTC val AUC score : ', roc_auc_score(val_y, pred_dtc))
print('ABC val AUC score : ', roc_auc_score(val_y, pred_abc))
print('RFC val AUC score : ', roc_auc_score(val_y, pred_rfc))
print('SC val AUC score : ', roc_auc_score(val_y, pred_sc))
print('MLPC val AUC score : ', roc_auc_score(val_y, pred_mlpc))
print()

# test model and report accuracy
model = AdaBoostClassifier()
model = model.fit(orig_train_X, orig_train_y)
pred = model.predict_proba(test_X)[: , 1]
print('ABC test AUC score : ', roc_auc_score(test_y, pred))
