import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def draw_graph(x_M, y_M, x_B, y_B, labelX, labelY):
    plt.figure(figsize=(7, 8))
    plt.title(" Распределение видов \n по двум признакам ")
    plt.scatter(x_M, y_M, marker='^', label="M")
    plt.scatter(x_B, y_B, marker='o', label="B")
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.legend()
    plt.grid()


data_frame = pd.get_dummies(pd.read_csv("breast_cancer.csv"), columns=["diagnosis"])
tags_knn = data_frame.get(["diagnosis_B", "diagnosis_M"]).to_numpy()
tags_lr = data_frame.get("diagnosis_B").to_numpy()

# Построение графиков по двум признакам
# data_M = data_frame[data_frame.diagnosis_M == 1]
# data_B = data_frame[data_frame.diagnosis_B == 1]
# field1 = "texture_mean"
# field2 = "area_mean"
#
# draw_graph(data_M.get([field1]), data_M.get([field2]),
#            data_B.get([field1]), data_B.get([field2]),
#            field1, field2)

# Неплохое сочетание признаков
# field1 = "radius_worst"
# field2 = "texture_worst"
# draw_graph(data_M.get([field1]), data_M.get([field2]),
#            data_B.get([field1]), data_B.get([field2]),
#            field1, field2)
# plt.show()

data_frame.drop(["Unnamed: 32", "diagnosis_B", "diagnosis_M", "id"], axis=1, inplace=True)


def knNeighbors(data_set, answers, isScaler):
    number_of_neighbors = list()
    predict_test_set = list()
    predict_train_set = list()
    average_value = list()

    if isScaler:
        scaler = StandardScaler()
        data = scaler.fit_transform(data_set)
    else:
        data = data_set.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(data, answers, test_size=0.3, random_state=0)
    kf = KFold(n_splits=5, shuffle=True)

    for i in range(1, 50):
        knn = KNeighborsClassifier(i)
        # for train_index, test_index in kf.split(data):
        #     knn.fit(data[train_index], answers[train_index])
        knn.fit(x_train, y_train)
        number_of_neighbors.append(i)
        predict_test_set.append(accuracy_score(y_test, knn.predict(x_test)))
        predict_train_set.append(accuracy_score(y_train, knn.predict(x_train)))
        array = cross_val_score(knn, x_test, y_test, cv=kf, scoring='accuracy')
        average = sum(array) / len(array)
        average_value.append(average)
    print(f"Оптимальное значение количества соседей = {predict_test_set.index(max(predict_test_set)) + 1}")
    print(f"Значение точности классификации при этом {max(predict_test_set)}")

    plt.subplot(2, 1, 1)
    plt.plot(number_of_neighbors, predict_test_set, label="Тестовые данные")
    plt.plot(number_of_neighbors, predict_train_set, label="Тренировочные данные")
    plt.title("Зависимость точности от количества соседей")
    plt.ylabel("p - точность")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(number_of_neighbors, average_value)
    plt.grid()
    plt.ylabel("average - среднее значение")
    plt.xlabel("n - количество соседей")
    plt.show()


def logisticRegression(data_set, answers, isScaler):
    list_of_c = list()
    result_test = list()
    result_train = list()
    average_value = list()

    if isScaler:
        scaler = StandardScaler()
        data = scaler.fit_transform(data_set)
    else:
        data = data_set.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data, answers, test_size=0.3, random_state=0)

    kf = KFold(n_splits=5, shuffle=True)

    for i in np.arange(0.01, 1, 0.01):
        lr = LogisticRegression(C=i)
        # for train_index, test_index in kf.split(data):
        #     lr.fit(data[train_index], answers[train_index])
        lr.fit(x_train, y_train)
        list_of_c.append(i)
        result_test.append(accuracy_score(y_test, lr.predict(x_test)))
        result_train.append(accuracy_score(y_train, lr.predict(x_train)))
        array = cross_val_score(lr, x_test, y_test, cv=kf, scoring='accuracy')
        average = sum(array) / len(array)
        average_value.append(average)
    print(f"Оптимальное значение параметра С = {list_of_c[result_test.index(max(result_test))]:1.2f}")
    print(f"Значение точности классификации при этом {max(result_test)}:1.2f")
    plt.subplot(2, 1, 1)
    plt.plot(list_of_c, result_test, label="Тестовая выборка")
    plt.plot(list_of_c, result_train, label="Тренировочная выборка")
    plt.title("Зависимость вероятности от параметра C")
    plt.ylabel("p - значение вероятности ")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(list_of_c, average_value)
    plt.ylabel("average - среднее значение ")
    plt.xlabel("С - параметр регуляризации ")
    plt.grid()
    plt.show()


knNeighbors(data_frame, tags_knn, False)
logisticRegression(data_frame, tags_lr, False)
knNeighbors(data_frame, tags_knn, True)
logisticRegression(data_frame, tags_lr, True)
