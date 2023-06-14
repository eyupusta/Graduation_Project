import numpy as np 
import pandas as pd 
from parsearff import parse_arff
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.utils import shuffle 
from collections import Counter 

# @attribute having_IP_Address  { -1,1 }
# @attribute URL_Length   { 1,0,-1 }
# @attribute Shortining_Service { 1,-1 }
# @attribute having_At_Symbol   { 1,-1 }
# @attribute double_slash_redirecting { -1,1 }
# @attribute Prefix_Suffix  { -1,1 }
# @attribute having_Sub_Domain  { -1,0,1 }
# @attribute SSLfinal_State  { -1,1,0 }
# @attribute Domain_registeration_length { -1,1 }
# @attribute Favicon { 1,-1 }
# @attribute port { 1,-1 }
# @attribute HTTPS_token { -1,1 }
# @attribute Request_URL  { 1,-1 }
# @attribute URL_of_Anchor { -1,0,1 }
# @attribute Links_in_tags { 1,-1,0 }
# @attribute SFH  { -1,1,0 }
# @attribute Submitting_to_email { -1,1 }
# @attribute Abnormal_URL { -1,1 }
# @attribute Redirect  { 0,1 }
# @attribute on_mouseover  { 1,-1 }
# @attribute RightClick  { 1,-1 }
# @attribute popUpWidnow  { 1,-1 }
# @attribute Iframe { 1,-1 }
# @attribute age_of_domain  { -1,1 }
# @attribute DNSRecord   { -1,1 }
# @attribute web_traffic  { -1,0,1 }
# @attribute Page_Rank { -1,1 }
# @attribute Google_Index { 1,-1 }
# @attribute Links_pointing_to_page { 1,0,-1 }
# @attribute Statistical_report { -1,1 }
# @attribute Result  { -1,1 }

class NaiveBayesClassifier:
	def __init__(self):
		pass

	def generate_freq_table(self):
		#  -1 +1
		#-1 #  #
		# 0 #  #
		#+1 #  #
		self.freqlist = []
		for i in range(self.features.shape[1]):
			self.freqlist.append(np.ones((np.unique(self.features[:,i]).shape[0],np.unique(self.labels).shape[0])))

		for i in range(self.features.shape[0]):
			for k in range(self.features.shape[1]):
				if self.freqlist[k].shape[0] == 3:
					if self.features[i][k] == -1:
						if self.labels[i] == -1:
							self.freqlist[k][0][0] +=1
						elif self.labels[i] == 1:
							self.freqlist[k][0][1] +=1
					elif self.features[i][k] == 0:
						if self.labels[i] == -1:
							self.freqlist[k][1][0] +=1
						elif self.labels[i] == 1:
							self.freqlist[k][1][1] +=1
					elif self.features[i][k] == 1:
						if self.labels[i] == -1:
							self.freqlist[k][2][0] +=1
						elif self.labels[i] == 1:
							self.freqlist[k][2][1] +=1
					else:
						continue
				else:
					if self.features[i][k] == -1:
						if self.labels[i] == -1:
							self.freqlist[k][0][0] +=1
						elif self.labels[i] == 1:
							self.freqlist[k][0][1] +=1
					elif self.features[i][k] == 1:
						if self.labels[i] == -1:
							self.freqlist[k][1][0] +=1
						elif self.labels[i] == 1:
							self.freqlist[k][1][1] +=1
						else:
							continue

	def fit(self, x, y):
		self.features = x
		self.labels = y
		self.generate_freq_table()

	def predict(self, x):
		#  -1 +1
		#-1 #  #
		# 0 #  #
		#+1 #  #
		classes, counts = np.unique(self.labels,return_counts=True)
		pos, neg, lenpos, lenneg = 0, 0, 0, 0
		if classes[0] == 1:
			lenpos, lenneg = counts[0], counts[1]
			pos = counts[0] / len(self.labels)
			neg = counts[1] / len(self.labels)
		else:
			lenpos, lenneg = counts[1], counts[0]
			pos = counts[1] / len(self.labels)
			neg = counts[0] / len(self.labels)

		self.pred_pos = 1
		self.pred_neg = 1
		for i in range(len(x)):
			if self.freqlist[i].shape[0] == 3:
				if x[i] == -1:
					self.pred_pos *= self.freqlist[i][0][1] / lenpos
					self.pred_neg *= self.freqlist[i][0][0] / lenneg
				elif x[i] == 0:
					self.pred_pos *= self.freqlist[i][1][1] / lenpos
					self.pred_neg *= self.freqlist[i][1][0] / lenneg
				elif x[i] == 1:
					self.pred_pos *= self.freqlist[i][2][1] / lenpos
					self.pred_neg *= self.freqlist[i][2][0] / lenneg
				else:
					pass
			else:
				if x[i] == -1:
					self.pred_pos *= self.freqlist[i][0][1] / lenpos
					self.pred_neg *= self.freqlist[i][0][0] / lenneg
				elif x[i] == 1:
					self.pred_pos *= self.freqlist[i][1][1] / lenpos
					self.pred_neg *= self.freqlist[i][1][0] / lenneg
				else:
					pass
		self.pred_pos *= pos
		self.pred_neg *= neg
		#print(self.pred_neg,self.pred_pos)

		return 1.0 if self.pred_pos > self.pred_neg else -1.0


data, columns = parse_arff('Training Dataset.arff')
df = pd.DataFrame(data, columns = columns, dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],df[df.columns[-1]])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = y_train.reshape(y_train.shape[0],1)
#y_test = y_test.reshape(y_test.shape[0],)

svm = NaiveBayesClassifier()

svm.fit(x_train, y_train)

l = []
for i in range(len(x_test)):
	l.append(svm.predict(x_test[i]))
print(y_test.shape)
print(list(y_test))
print("********************************")
print(l)