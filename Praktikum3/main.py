import pandas as pd
import numpy as np


car = pd.read_csv("data/car.csv")
car.head()


x = car.drop(["acceptabillity"], axis=1 )
x.head()
y = car["acceptabillity"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

from sklearn.naive_bayes import GaussianNB
#memanggil fungsi naive bayes
modelnb = GaussianNB()
#memasukkan data training pada klasifikasi naive bayes
nbtrain = modelnb.fit(x_train, y_train)

y_pred = nbtrain.predict(x_test)
y_pred

np.array(y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import _classification_report
print(_classification_report(y_test, y_pred))