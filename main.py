import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("streamlit example")
st.write("""
# Example different classifier
which one is the best?
""")
dataset_name = st.sidebar.selectbox("select dataset",("Iris","breast cancer","wine dataset"))
#st.write(datasets_name)
classifier_name = st.sidebar.selectbox("select classifier",("KNN","SVM","random forest"))
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "breast cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y))) 
fig = plt.figure()
sns.boxplot(data=X, orient='h')

st.pyplot()

#plottinhg a histogram

plt.hist(X)

st.pyplot()


 #BUILDING OUR ALGORITHM

# #define a function necessary for our algorithms
def add_parameter_ui(name_of_clf):
    params=dict()
    if name_of_clf == "KNN":
        k= st.sidebar.slider("K",1,15)
    else:
        name_of_clf =="SVM"
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
       
    return params 

params=add_parameter_ui(classifier_name)


def get_classifier(name_of_clf,params):
    if name_of_clf=="KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif name_of_clf=="SVM":
          clf = SVC(C=params["C"])
    else:

        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)

    return clf

clf = get_classifier(classifier_name, params)


#classification

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

pca = PCA(2)
x_projected = pca.fit_transform(X)
x1 = x_projected[:,0]
x2 = x_projected[:,1]
fit = plt.figure()
plt.scatter(x1,x2,c=y, alpha=0.8,cmap="wridis")
plt.xlable["principal component 1"]
plt.ylable["principal component 2"]
plt.colorbar()
#plt.show()
st.pyplot()