import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("train_large.csv")
df = df.iloc[: , 1:]
df.insert(0, '', 1)
x_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

param = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
list = []
size = int(x_train.shape[0]/10)

for p in param:
	int = 0
	for i in range(10):
		a_train = (x_train.iloc[:i*size, :]).append(x_train.iloc[(i+1)*size:, :])
		a_test = x_train.iloc[i*size:(i+1)*size, :]
		b_train = (y_train[:i*size]).append(y_train.iloc[(i+1)*size:])
		b_test = y_train[i*size:(i+1)*size]
	
		x = np.transpose(a_train)
		y = np.dot(x, a_train)
		y = np.linalg.inv(y + p*np.identity(a_train.shape[1]))
		y = np.dot(y, x)
		y = np.dot(y, b_train)
		y = np.dot(a_test, y)

		int = int + np.sum(np.square(b_test - y))/np.sum(np.square(y))
	int = int/10
	list.append(int)

best = param[list.index(min(list))]

model = LassoLars(alpha=best)
model.fit(x_train, y_train)
weights = model.coef_.tolist()
list1 = [weights.index(w) for w in weights if w>0]
list2 = [weights.index(w) for w in weights if w<0]
x_train1 = x_train.iloc[:, list1]
x_train2 = x_train.iloc[:, list2]

poly = PolynomialFeatures(degree=2)
x_train1 = poly.fit_transform(x_train1)

model = LassoLars(alpha=best)
model.fit(x_train1, y_train)
weights = model.coef_.tolist()
list1 = [weights.index(w) for w in weights if w!=0]
x_train1 = x_train1[:, list1]

poly = PolynomialFeatures(degree=2)
x_train2 = poly.fit_transform(x_train2)

model = LassoLars(alpha=best)
model.fit(x_train2, y_train)
weights = model.coef_.tolist()
list2 = [weights.index(w) for w in weights if w!=0]

x_train2 = x_train2[:, list2]
x_train = np.append(x_train1, x_train2, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

x = np.transpose(x_train)
y = np.dot(x, x_train)
y = np.linalg.inv(y)
y = np.dot(y, x)
y = np.dot(y, y_train)
z = np.dot(x_test, y)

score = r2_score(y_test, z)
print(score)


