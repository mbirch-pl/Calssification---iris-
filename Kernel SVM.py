import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

data = load_iris()

X = data['data']
y = data['target']

X,y = shuffle(X,y,random_state=0)

from  sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

print(acc,'\n',cm)