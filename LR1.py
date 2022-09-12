import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data_frame = pd.get_dummies(pd.read_csv("breast_cancer.csv"), columns=["diagnosis"]).drop(columns=["id", "Unnamed: 32"])

x_train, x_test, y_train, y_test = train_test_split(data_frame.get(data_frame.columns[:-2]),
                                                    data_frame.get(data_frame.columns[:-3:-1]),
                                                    test_size=0.10, random_state=0)

number_of_neighbors = list()
result_of_predict = list()
result_of_predictTrain = list()
for i in range(2, 50):
    knn = KNeighborsClassifier(i)
    knn.fit(x_train, y_train)
    number_of_neighbors.append(i)
    result_of_predict.append(accuracy_score(y_test, knn.predict(x_test)))
    result_of_predictTrain.append(accuracy_score(y_train, knn.predict(x_train)))

plt.subplot(1, 1, 1)
plt.plot(number_of_neighbors, result_of_predict, label="Тестовые данные")
plt.plot(number_of_neighbors, result_of_predictTrain, label="Тренировочные данные")
plt.title("Зависимость точности от количества соседей")
plt.xlabel("n - количество соседей")
plt.ylabel("p - точность")
plt.legend()
plt.grid()
plt.show()

