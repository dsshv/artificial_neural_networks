import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import math
#прочитать обработанную дату
ds = pd.read_csv('KmeansData_New.csv')
df = pd.DataFrame(ds)
#удалить столбцы без названия (если есть)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# перегнать датафрейм в матрицу для PCA
X1 = df.values #[:, 0:685]
# PCA на 2 два компонента
pca = PCA(n_components=2)
fit = pca.fit(X1)
features = fit.transform(X1)

#Kmeans на 4 кластера
Kmean = KMeans(n_clusters=4)

#расстояние от точки до центра кластера
def dist(x, y, cluster_centers):
    distance = []

    for (cx, cy) in cluster_centers:
        temp = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        distance.append(temp)

    return distance


def k_mean_distance(data, cluster_centers):
    distances_p = []

    for (x, y) in data:
        distance = dist(x, y, cluster_centers)
        distances_p.append(distance)
    return distances_p

#разбор кластеров для плота
label = Kmean.fit_predict(features)

label0 = features[label == 0]
label1 = features[label == 1]
label2 = features[label == 2]
label3 = features[label == 3]

#Плот кластеров
centres = Kmean.cluster_centers_
print(centres)
plt.scatter(label0[:, 0], label0[:, 1], s=5, color='blue')  # 0 много точек - гидриды
plt.scatter(label1[:, 0], label1[:, 1], s=10, color='black')
plt.scatter(label2[:, 0], label2[:, 1], s=5, color='g')  # 2 кластер - гидрид
plt.scatter(label3[:, 0], label3[:, 1], s=5, color='y')  # 3 кластер - гидрид
plt.scatter(centres[0, 0], centres[0, 1], s=10, color='r')
plt.scatter(centres[1, 0], centres[1, 1], s=10, color='r')
plt.scatter(centres[2, 0], centres[2, 1], s=10, color='r')
plt.scatter(centres[3, 0], centres[3, 1], s=10, color='r')
plt.show()

#расчет растояний до центров
mean_distance = k_mean_distance(features, centres)

k_df = pd.DataFrame(mean_distance)
print(k_df.head())


# гидрид - 3 слева
#
# 2 кластер - гидрид (зеленый)
# 3 кластер - оксид (желтый)
# 0 кластер - карбид (синий)
# металлы записывает в гидриды (зеленые точки)
#
# print(features[label==1])
