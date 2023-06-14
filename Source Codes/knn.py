import numpy as np 
import pandas as pd 
from parsearff import parse_arff
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.utils import shuffle 
from collections import Counter 

class KNN:
	def __init__(self, n_neighbors = 3):
		self.n_neighbors = n_neighbors

	def fit(self, features, labels, formula = 'euclidean'):
		self.features = features
		self.labels = labels
		self.formula = formula
		self.distances = {}
		self.distance = []

	def euclidean(self, feature, y):
		#sqrt(sum((x - y)^2))
		return np.linalg.norm(np.array(feature)-np.array(y))


	def predict(self, x_test):
		if self.formula == 'euclidean':
			for feature, index in zip(self.features,np.arange(len(self.features))):
				self.distance.append([self.euclidean(feature,x_test), self.labels[index][0]])

			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]

			return prediction

		elif self.formula == 'manhattan':
			for feature, index in zip(self.features,len(self.features)):
				self.distance.append([manhattan(feature,x_test),self.labels[index]])

			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]

			return prediction

		elif self.formula == 'chebsyhey':
			for feature, index in zip(self.features,len(self.features)):
				self.distance.append([chebsyhey(feature,x_test),self.labels[index]])
			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]

			return prediction

		elif self.formula == 'minkowski':
			for feature, index in zip(self.features,len(self.features)):
				self.distance.append([minkowski(feature,x_test, 1),self.labels[index]])

			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]
			return prediction

		elif self.formula == 'wminkowski':
			for feature, index in zip(self.features,len(self.features)):
				self.distance.append([wminkowski(feature,x_test, 1, np.ones(feature.shape)),self.labels[index]])

			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]
			return prediction

		elif self.formula == 'seuclidean':
			for feature, index in zip(self.features,len(self.features)):
				self.distance.append([seuclidean(feature,x_test, 2),self.labels[index]])

			votes = [i[1] for i in sorted(self.distance)[:self.n_neighbors]]
			prediction = Counter(votes).most_common(1)[0][0]
			return prediction



	
	def manhattan(self, feature, y):
		#sum(|x - y|)
		return np.sum(np.abs(np.array(feature)-np.array(y)))

	def chebsyhey(self, feature, y):
		#max(|x - y|)
		return np.max(np.abs(np.array(feature)-np.array(y)))

	def minkowski(self, feature, y, p):
		#sum(|x - y|^p)^(1/p)
		return np.pow(np.sum(np.pow(np.abs(np.array(feature)-np.array(y))), p), 1/p)


	def wminkowski(self, feature, y, p, w):
		#sum(|w * (x - y)|^p)^(1/p)
		return np.pow(np.pow(np.abs(np.dot(w,np.array(feature)-np.array(y))), p),1/p)
		
	def seuclidean(self, feature, y, V):
		#sqrt(sum((x - y)^2 / V))
		return np.sqrt(np.pow(np.sum(np.array(feature)-np.array(y)),2) / V)



knn = KNN()
data, columns = parse_arff('Training Dataset.arff')
df = pd.DataFrame(data, columns = columns, dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],df[df.columns[-1]])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

knn.fit(x_test, y_test)

print(knn.predict(x_train[0]))
print(y_train[0])