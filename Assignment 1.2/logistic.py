import numpy as np
import pandas as pd
import math
import sys
import time


start = time.time()

train_path = sys.argv[2]
test_path = sys.argv[3]

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

if sys.argv[1] == 'a':
	param = open(sys.argv[4], 'r').readlines()
	strat = int(param[0])
	if strat == 3:
		l_rate = float(param[1].split(',')[0])
		alpha = float(param[1].split(',')[1])
		beta = float(param[1].split(',')[2])
	else:
		l_rate = float(param[1])
	iter = int(param[2])
	
	X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
	X_test = np.c_[np.ones([X_test.shape[0], 1]), X_test]
	weight = np.zeros([X_train.shape[1], 8])

	if strat == 1:
		for i in range(iter):
			x = np.transpose(np.exp(np.dot(X_train, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train), y - y_train)/X_train.shape[0]
			weight = weight - z * l_rate
			
			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	elif strat == 2:
		for i in range(iter):
			x = np.transpose(np.exp(np.dot(X_train, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train), y - y_train)/X_train.shape[0]
			weight = weight - z * l_rate/math.sqrt(i+1)

			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	elif strat == 3:
		for i in range(iter):
			l_rate1 = l_rate
			x = np.transpose(np.exp(np.dot(X_train, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train), y - y_train)/X_train.shape[0]
			l = np.sum(-1 * y_train * np.log(np.clip(y, 1e-15, 1-1e-15)), axis=1)
			w = np.sum(l) / l.shape[0]

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

			weight = weight - z * l_rate1 

			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	pred = []
	for j in np.dot(X_test, weight).tolist():
		pred.append(len_of_stay[j.index(max(j))])

	np.savetxt(sys.argv[5], pred)
	np.savetxt(sys.argv[6], weight, delimiter='\n')


if sys.argv[1] == 'b':
	param = open(sys.argv[4], 'r').readlines()
	strat = int(param[0])
	if strat == 3:
		l_rate = float(param[1].split(',')[0])
		alpha = float(param[1].split(',')[1])
		beta = float(param[1].split(',')[2])
	else:
		l_rate = float(param[1])
	iter = int(param[2])
	size = int(param[3])
	
	X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
	X_test = np.c_[np.ones([X_test.shape[0], 1]), X_test]

	batch = []
	for i in range(0, X_train.shape[0], size):
		j = i + size
		if j > X_train.shape[0]:
			break
		batch.append((X_train[i:j], y_train[i:j]))

	weight = np.zeros([X_train.shape[1], 8])

	if strat == 1:
		for i in range(iter):
			for (X_train1, y_train1) in batch:
				x = np.transpose(np.exp(np.dot(X_train1, weight)))
				y = np.transpose(x/np.sum(x, axis=0))
				z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]      
				weight = weight - z * l_rate
			
			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	elif strat == 2:
		for i in range(iter):
			for (X_train1, y_train1) in batch:
				x = np.transpose(np.exp(np.dot(X_train1, weight)))
				y = np.transpose(x/np.sum(x, axis=0))
				z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]
				weight = weight - z * l_rate/math.sqrt(i+1)

			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	elif strat == 3:
		for i in range(iter):
			l_rate1 = l_rate
			x = np.transpose(np.exp(np.dot(X_train, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train), y - y_train)/X_train.shape[0]
			l = np.sum(-1 * y_train * np.log(np.clip(y, 1e-15, 1-1e-15)), axis=1)
			w = np.sum(l) / l.shape[0]

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

			if (i+1)%50 == 0:
				pred = []
				for j in np.dot(X_test, weight).tolist():
					pred.append(len_of_stay[j.index(max(j))])
				np.savetxt(sys.argv[5], pred)
				np.savetxt(sys.argv[6], weight, delimiter='\n')

	pred = []
	for j in np.dot(X_test, weight).tolist():
		pred.append(len_of_stay[j.index(max(j))])

	np.savetxt(sys.argv[5], pred)
	np.savetxt(sys.argv[6], weight, delimiter='\n')


if sys.argv[1] == 'c':
	strat = 2
	l_rate = 5
	iter = 2000
	size = 50
	
	X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
	X_test = np.c_[np.ones([X_test.shape[0], 1]), X_test]

	batch = []
	for i in range(0, X_train.shape[0], size):
		j = i + size
		if j > X_train.shape[0]:
			break
		batch.append((X_train[i:j], y_train[i:j]))

	weight = np.zeros([X_train.shape[1], 8])

	for i in range(iter):
		for (X_train1, y_train1) in batch:
			x = np.transpose(np.exp(np.dot(X_train1, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]
			weight = weight - z * l_rate/math.sqrt(i+1)

		if (i+1)%25 == 0:
			pred = []
			for j in np.dot(X_test, weight).tolist():
				pred.append(len_of_stay[j.index(max(j))])
			np.savetxt(sys.argv[4], pred)
			np.savetxt(sys.argv[5], weight, delimiter='\n')

	pred = []
	for j in np.dot(X_test, weight).tolist():
		pred.append(len_of_stay[j.index(max(j))])

	np.savetxt(sys.argv[4], pred)
	np.savetxt(sys.argv[5], weight, delimiter='\n')


if sys.argv[1] == 'd':
	strat = 2
	l_rate = 5
	iter = 5000
	size = 50
	
	X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
	X_test = np.c_[np.ones([X_test.shape[0], 1]), X_test]

	batch = []
	for i in range(0, X_train.shape[0], size):
		j = i + size
		if j > X_train.shape[0]:
			break
		batch.append((X_train[i:j], y_train[i:j]))

	weight = np.zeros([X_train.shape[1], 8])

	for i in range(iter):
		for (X_train1, y_train1) in batch:
			x = np.transpose(np.exp(np.dot(X_train1, weight)))
			y = np.transpose(x/np.sum(x, axis=0))
			z = np.dot(np.transpose(X_train1), y - y_train1)/X_train1.shape[0]
			weight = weight - z * l_rate/math.sqrt(i+1)

		if (i+1)%25 == 0:
			pred = []
			for j in np.dot(X_test, weight).tolist():
				pred.append(len_of_stay[j.index(max(j))])
			np.savetxt(sys.argv[4], pred)
			np.savetxt(sys.argv[5], weight, delimiter='\n')

	pred = []
	for j in np.dot(X_test, weight).tolist():
		pred.append(len_of_stay[j.index(max(j))])

	np.savetxt(sys.argv[4], pred)
	np.savetxt(sys.argv[5], weight, delimiter='\n')


end = time.time()
#print(f"Runtime of the program is {end - start}")
	
