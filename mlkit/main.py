from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlkit.classification import LogisticRegression

data = load_breast_cancer()
X = data.data
y = data.target
y = y.reshape(-1,1)
X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.train(X_train,y_train,iterations=500,verbose=True)
y_pred = model.predict(X_test)
cost_history = model.cost_log
