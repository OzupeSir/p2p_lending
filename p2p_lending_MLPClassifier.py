import sys
import pandas as pd
import numpy as np


data = pd.read_csv("p2p_lending.csv")

features=data[data.columns[5:18]]
label=data[['安全标']]


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=0)


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5), random_state=1)
mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(score)

sex = pd.get_dummies(data['性别'], prefix='sex')
edu_degree = pd.get_dummies(data['最高学历'], prefix='edudg')
edu_type = pd.get_dummies(data['大学类型'], prefix='eduty')
loan_type = pd.get_dummies(data['标准化借款类型'], prefix='loan')
assured = pd.get_dummies(data['担保'], prefix='assure')

features= pd.concat([sex, edu_degree, edu_type, loan_type, assured, features], axis=1)
#print(features)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
features= pd.DataFrame(features)
#print(features)


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=0)


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10), random_state=1)
mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(score)