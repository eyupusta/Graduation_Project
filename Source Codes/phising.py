import numpy as np 
import pandas as pd 
from parsearff import parse_arff
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.utils import shuffle 
from collections import Counter
from parsearff import parse_arff
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


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


data, columns = parse_arff('Training Dataset.arff')
df = pd.DataFrame(data, columns = columns, dtype=np.float64)
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],df[df.columns[-1]])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'mahalanobis', 'seuclidean']


knn = []
for k in range(1, 101):
    for weight in weights:
        for algo in algorithm:
            for m in metric:
                knn.append(KNeighborsClassifier(n_neighbors=k, weights = weight, algorithm = algo).fit(x_train, y_train))

knnacc = []
for model in knn:
    knn.append(accuracy_score(y_test, model.predict(x_test)))



kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C = [1, 2, 3, 4, 5] # regularization parameter
degree = [1, 2, 3, 4, 5, 6] # polinomial degree
gamma = ['scale', 'auto']
decision_function_shape = ['ovo', 'ovr']
#SVC

models = []
for kernel in kernels:
    for c in C:
        for deg in degree:
            for gam in gamma:
                for decision in decision_function_shape:
                    models.append(SVC(kernel = kernel, C = c, degree = deg, gamma = gam).fit(x_train, y_train))


losses = ['hinge', 'squared_hinge']
penalty = ['l2']
C = [1, 2, 3, 4, 5] # regularization parameter
multi_class = ['ovr', 'crammer_singer']
#LİNEARSVC


linearmodel = []
for loss in losses:
    for pen in penalty:
        for c in C:
            for multi in multi_class:
                linearmodel.append(LinearSVC(loss = loss, penalty = pen, C = c, multi_class = multi).fit(x_train, y_train))



acc_svc = []
for model in models:
    acc_svc.append(accuracy_score(y_test, model.predict(x_test)))



acc_linearsvc = []
for model in linearmodel:
    acc_linearsvc.append(accuracy_score(y_test, model.predict(x_test)))