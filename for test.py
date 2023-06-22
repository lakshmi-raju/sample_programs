from KNN import KNN
import numpy as np

x_train = np.array([[1,2],[1,3],[1,4],[5,1],[1,5],[8,9],[7,9],[10,9],[11,9]])
y_train =np.array([1,1,1,1,1,2,2,2,2])
model = KNN(3)
model.fit(x_train,y_train)
model.predict(np.array([[0,1]]))
# y = [[0,1]]
# from math import dist
# #print([dist(x_train[:i],y) for i in range(len(x_train)) ])
# print(len(x_train[8,1]))
#delete new line