import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys


train_path = sys.argv[1]
test_path = sys.argv[2]

train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])

train = train.drop(columns = ['Length of Stay'])

#Ensuring consistency of One-Hot Encoding

data = pd.concat([train, test], ignore_index = True)
cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
X_train = data[:train.shape[0], :]
X_test = data[train.shape[0]:, :]

len_of_stay = [1, 2, 3, 4, 5, 6, 7, 8]
y_train = pd.get_dummies(y_train)

iter = 100
	
X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
X_test = np.c_[np.ones([X_test.shape[0], 1]), X_test]
weight = np.zeros([X_train.shape[1], 8])
l1 = []
l2 = []
l3 = []

strat = 1
l_rates = [0.5, 1, 1.5, 1.6, 1.7, 2, 2.5]
sizes = [50, 100, 250, 400, 500, 750, 1000]
if strat == 1:
	for size in sizes:
		batch = []
		for i in range(0, X_train.shape[0], size):
			j = i + size
			if j > X_train.shape[0]:
				break
			batch.append((X_train[i:j], y_train[i:j]))

		for l_rate in l_rates:
			l1 = []
			weight = np.zeros([X_train.shape[1], 8])
			for i in range(iter):
				x = np.transpose(np.exp(np.dot(X_train, weight)))
				y = np.transpose(x/np.sum(x, axis=0))
				l = np.sum(-1 * y_train * np.log(np.clip(y, 1e-15, 1-1e-15)), axis=1)
				w = np.sum(l) / l.shape[0]
				l1.append(w)

				for (X_train1, y_train1) in batch:
					x = np.transpose(np.exp(np.dot(X_train1, weight)))
					y = np.transpose(x/np.sum(x, axis=0))
					z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]      
					weight = weight - z * l_rate
		
			plt.plot([i for i in range(iter)], l1, label=str(l_rate)+'  '+str(size))

weight = np.zeros([X_train.shape[1], 8])
strat = 2
l_rates = [0.5, 1, 2, 3, 4, 5]
sizes = [50, 100, 200, 400, 800]
if strat == 2:
	for size in sizes:
		batch = []
		for i in range(0, X_train.shape[0], size):
			j = i + size
			if j > X_train.shape[0]:
				break
			batch.append((X_train[i:j], y_train[i:j]))

		for l_rate in l_rates:
			l2 = []
			weight = np.zeros([X_train.shape[1], 8])
			for i in range(iter):
				x = np.transpose(np.exp(np.dot(X_train, weight)))
				y = np.transpose(x/np.sum(x, axis=0))
				l = np.sum(-1 * y_train * np.log(np.clip(y, 1e-15, 1-1e-15)), axis=1)
				w = np.sum(l) / l.shape[0]
				l2.append(w)

				for (X_train1, y_train1) in batch:
					x = np.transpose(np.exp(np.dot(X_train1, weight)))
					y = np.transpose(x/np.sum(x, axis=0))
					z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]
					weight = weight - z * l_rate/math.sqrt(i+1)

			plt.plot([i for i in range(iter)], l2, label=str(l_rate)+'  '+str(size))

weight = np.zeros([X_train.shape[1], 8])
strat = 3
l_rates = [0.5, 1, 1.5, 2, 2.5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sizes = [50, 100, 200, 400, 800]
if strat == 3:
	for size in sizes:
		batch = []
		for i in range(0, X_train.shape[0], size):
			j = i + size
			if j > X_train.shape[0]:
				break
			batch.append((X_train[i:j], y_train[i:j]))

		for l_rate in l_rates:
			for alpha in alphas:
				for beta in betas:
					l3 = []
					weight = np.zeros([X_train.shape[1], 8])
					for i in range(iter):
						l_rate1 = l_rate
						x = np.transpose(np.exp(np.dot(X_train, weight)))
						y = np.transpose(x/np.sum(x, axis=0))
						z = np.dot(np.transpose(X_train), y - y_train)/X_train.shape[0]
						l = np.sum(-1 * y_train * np.log(np.clip(y, 1e-15, 1-1e-15)), axis=1)
						w = np.sum(l) / l.shape[0]
						l3.append(w)

						x1 = np.transpose(np.exp(np.dot(X_train, weight - l_rate1*z)))
						y1 = np.transpose(x1/np.sum(x1, axis=0))
						l1 = np.sum(-1 * y_train * np.log(np.clip(y1, 1e-15, 1-1e-15)), axis=1)
						w1 = np.sum(l1) / l1.shape[0]
					
						while (w1 > w - alpha*l_rate1*(np.linalg.norm(z)**2)):
							l_rate1 = l_rate1 * beta
							x1 = np.transpose(np.exp(np.dot(X_train, weight - l_rate1*z)))
							y1 = np.transpose(x1/np.sum(x1, axis=0))
							l1 = np.sum(-1 * y_train * np.log(np.clip(y1, 1e-15, 1-(1e-15))), axis=1)
							w1 = np.sum(l1) / l1.shape[0]

						for (X_train1, y_train1) in batch:
							x = np.transpose(np.exp(np.dot(X_train1, weight)))
							y = np.transpose(x / np.sum(x, axis=0))
							z = np.dot(np.transpose(X_train1), y - y_train1) / X_train1.shape[0]
							weight = weight - z * l_rate1 	

					plt.plot([i for i in range(iter)], l3, label=str(size))

#plt.plot([i for i in range(iter)], l1, label ='1')
#plt.plot([i for i in range(iter)], l2, label ='2')
#plt.plot([i for i in range(iter)], l3, label ='3')
plt.grid()
plt.legend(bbox_to_anchor=(1.04,0.5))
plt.show()

