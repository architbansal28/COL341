import numpy as np
import pandas as pd
import math
import sys

from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(sys.argv[2])
df = df.iloc[: , 1:]
df.insert(0, '', 1)
x_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

df = pd.read_csv(sys.argv[3])
df = df.iloc[: , 1:]
df.insert(0, '', 1)
x_test = df


if sys.argv[1]=='a':
	x = np.transpose(x_train)
	y = np.dot(x, x_train)
	y = np.linalg.inv(y)
	y = np.dot(y, x)
	y = np.dot(y, y_train)
	z = np.dot(x_test, y)

	np.savetxt(sys.argv[4], z)
	np.savetxt(sys.argv[5], y)

elif sys.argv[1]=='b':
	param = np.loadtxt(sys.argv[4], delimiter=',')
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

	x = np.transpose(x_train)
	y = np.dot(x, x_train)
	y = np.linalg.inv(y + best*np.identity(x_train.shape[1]))
	y = np.dot(y, x)
	y = np.dot(y, y_train)
	z = np.dot(x_test, y)

	np.savetxt(sys.argv[5], z)
	np.savetxt(sys.argv[6], y)
	with open(sys.argv[7], 'w') as f:
		f.write(str(best))


elif sys.argv[1]=='c':
	model = LassoLars(alpha=0.00005)
	model.fit(x_train, y_train)
	weights = model.coef_.tolist()
	list1 = [weights.index(w) for w in weights if w>0]
	list2 = [weights.index(w) for w in weights if w<0]

	x_train1 = x_train.iloc[:, list1]
	x_train2 = x_train.iloc[:, list2]
	x_test1 = x_test.iloc[:, list1]
	x_test2 = x_test.iloc[:, list2]

	poly = PolynomialFeatures(degree=2)
	x_train1 = poly.fit_transform(x_train1)
	x_test1 = poly.fit_transform(x_test1)

	model = LassoLars(alpha=0.00005)
	model.fit(x_train1, y_train)
	weights = model.coef_.tolist()
	list1 = [weights.index(w) for w in weights if w!=0]
	x_train1 = x_train1[:, list1]
	x_test1 = x_test1[:, list1]

	poly = PolynomialFeatures(degree=2)
	x_train2 = poly.fit_transform(x_train2)
	x_test2 = poly.fit_transform(x_test2)

	model = LassoLars(alpha=0.00005)
	model.fit(x_train2, y_train)
	weights = model.coef_.tolist()
	list2 = [weights.index(w) for w in weights if w!=0]
	x_train2 = x_train2[:, list2]
	x_test2 = x_test2[:, list2]
	x_train = np.append(x_train1, x_train2, axis=1)
	x_test = np.append(x_test1, x_test2, axis=1)
	
	x = np.transpose(x_train)
	y = np.dot(x, x_train)
	y = np.linalg.inv(y)
	y = np.dot(y, x)
	y = np.dot(y, y_train)
	z = np.dot(x_test, y)

	np.savetxt(sys.argv[4], z)
		

