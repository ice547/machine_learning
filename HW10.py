import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("mnist_train.csv")

x = dataset.iloc[:, 1:785].values
y = dataset.iloc[:,0].values

show_img = np.reshape(x[1,0:784],(28,28))
plt.matshow(show_img,cmap = plt.get_cmap('gray'))
plt.show

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.5 , random_state = 0 )

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state =0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

GitTest123
